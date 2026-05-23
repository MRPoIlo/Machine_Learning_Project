/* ============================================================
   Vanilla-JS dynamic table
   Features: search, filters, sort, pagination
   Usage: createDataTable({ target, dataUrl, columns, filters, pageSize })
   ============================================================ */

(function (global) {
    'use strict';

    function el(tag, attrs, children) {
        const node = document.createElement(tag);
        if (attrs) {
            for (const k in attrs) {
                if (k === 'class') node.className = attrs[k];
                else if (k === 'dataset') Object.assign(node.dataset, attrs[k]);
                else if (k.startsWith('on') && typeof attrs[k] === 'function') node.addEventListener(k.slice(2), attrs[k]);
                else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
            }
        }
        if (children) {
            (Array.isArray(children) ? children : [children]).forEach(c => {
                if (c == null) return;
                node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
            });
        }
        return node;
    }

    function compareValues(a, b, type) {
        if (a == null) return 1;
        if (b == null) return -1;
        if (type === 'number') return Number(a) - Number(b);
        return String(a).localeCompare(String(b), 'es', { sensitivity: 'base', numeric: true });
    }

    function formatCell(value, col) {
        if (value == null || (typeof value === 'number' && Number.isNaN(value))) return '—';
        if (col.format) return col.format(value);
        if (col.type === 'number' && typeof value === 'number') {
            return Number.isInteger(value) ? value.toString() : value.toFixed(2);
        }
        return String(value);
    }

    const RISK_LEVEL = {
        'No Risk': 1,
        'Low Risk': 2,
        'Medium Risk': 3,
        'High Risk': 4,
        'Sanitaryly not feasible': 5,
    };

    function renderRiskMeter(label) {
        const level = RISK_LEVEL[label] || 0;
        const wrap = document.createElement('span');
        wrap.className = 'risk-meter risk-meter--' + level;
        wrap.setAttribute('aria-label', label + ' (level ' + level + ' of 5)');

        const dots = document.createElement('span');
        dots.className = 'risk-meter__dots';
        dots.setAttribute('aria-hidden', 'true');
        for (let i = 1; i <= 5; i++) {
            const dot = document.createElement('span');
            dot.className = 'risk-meter__dot' + (i <= level ? ' is-filled' : '');
            dots.appendChild(dot);
        }

        const text = document.createElement('span');
        text.className = 'risk-meter__label';
        text.textContent = label;

        wrap.appendChild(dots);
        wrap.appendChild(text);
        return wrap;
    }

    function createDataTable(opts) {
        const root = typeof opts.target === 'string'
            ? document.querySelector(opts.target)
            : opts.target;
        if (!root) return;

        const columns  = opts.columns;
        const filters  = opts.filters || [];
        const pageSize = opts.pageSize || 25;

        const state = {
            data: [],
            view: [],
            sort: { key: null, dir: 'asc' },
            search: '',
            filterValues: {},
            page: 1,
        };

        // ---- DOM scaffolding ----
        root.classList.add('datatable');
        root.innerHTML = '';

        const toolbar = el('div', { class: 'datatable__toolbar' });

        const searchInput = el('input', {
            type: 'search',
            placeholder: 'Search municipality, department…',
            class: 'datatable__search',
            'aria-label': 'Search rows',
        });

        const filterGroup = el('div', { class: 'datatable__filters' });
        const filterSelects = {};
        filters.forEach(f => {
            const select = el('select', { class: 'datatable__filter', 'data-key': f.key, 'aria-label': f.label });
            select.appendChild(el('option', { value: '' }, `All ${f.label}`));
            filterSelects[f.key] = select;
            filterGroup.appendChild(select);
        });

        toolbar.appendChild(searchInput);
        if (filters.length) toolbar.appendChild(filterGroup);

        const status = el('div', { class: 'datatable__status' });

        const wrapper = el('div', { class: 'datatable__wrapper' });
        const table = el('table', { class: 'datatable__table' });
        const thead = el('thead');
        const headerRow = el('tr');
        columns.forEach(col => {
            const th = el('th', { class: col.sortable !== false ? 'is-sortable' : null, 'data-key': col.key });
            th.appendChild(document.createTextNode(col.label));
            if (col.sortable !== false) {
                const arrow = el('span', { class: 'datatable__sort-arrow' }, '↕');
                th.appendChild(arrow);
                th.addEventListener('click', () => toggleSort(col.key));
            }
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        const tbody = el('tbody');
        table.appendChild(thead);
        table.appendChild(tbody);
        wrapper.appendChild(table);

        const pagination = el('div', { class: 'datatable__pagination' });
        const prevBtn = el('button', { type: 'button', class: 'datatable__page-btn' }, '← Prev');
        const nextBtn = el('button', { type: 'button', class: 'datatable__page-btn' }, 'Next →');
        const pageInfo = el('span', { class: 'datatable__page-info' });
        prevBtn.addEventListener('click', () => { if (state.page > 1) { state.page--; render(); } });
        nextBtn.addEventListener('click', () => {
            if (state.page < totalPages()) { state.page++; render(); }
        });
        pagination.appendChild(prevBtn);
        pagination.appendChild(pageInfo);
        pagination.appendChild(nextBtn);

        const frame = el('div', { class: 'datatable__frame' });
        frame.appendChild(status);
        frame.appendChild(wrapper);
        frame.appendChild(pagination);

        root.appendChild(toolbar);
        root.appendChild(frame);

        // ---- behavior ----
        searchInput.addEventListener('input', () => {
            state.search = searchInput.value.trim().toLowerCase();
            state.page = 1;
            applyView();
        });

        Object.keys(filterSelects).forEach(key => {
            filterSelects[key].addEventListener('change', () => {
                state.filterValues[key] = filterSelects[key].value;
                state.page = 1;
                applyView();
            });
        });

        function toggleSort(key) {
            if (state.sort.key === key) {
                state.sort.dir = state.sort.dir === 'asc' ? 'desc' : 'asc';
            } else {
                state.sort.key = key;
                state.sort.dir = 'asc';
            }
            state.page = 1;
            applyView();
        }

        function totalPages() {
            return Math.max(1, Math.ceil(state.view.length / pageSize));
        }

        function applyView() {
            const colByKey = {};
            columns.forEach(c => { colByKey[c.key] = c; });

            let v = state.data;

            if (state.search) {
                const q = state.search;
                v = v.filter(row => columns.some(c => {
                    const val = row[c.key];
                    return val != null && String(val).toLowerCase().includes(q);
                }));
            }

            Object.keys(state.filterValues).forEach(key => {
                const val = state.filterValues[key];
                if (val === '' || val == null) return;
                v = v.filter(row => String(row[key]) === val);
            });

            if (state.sort.key) {
                const col = colByKey[state.sort.key];
                const dir = state.sort.dir === 'asc' ? 1 : -1;
                v = v.slice().sort((a, b) =>
                    dir * compareValues(a[state.sort.key], b[state.sort.key], col && col.type)
                );
            }

            state.view = v;
            render();
        }

        function render() {
            tbody.innerHTML = '';
            const start = (state.page - 1) * pageSize;
            const end = start + pageSize;
            const slice = state.view.slice(start, end);

            if (slice.length === 0) {
                const emptyTr = el('tr', { class: 'datatable__empty-row' });
                emptyTr.appendChild(el('td', { colspan: columns.length }, 'No rows match your filters.'));
                tbody.appendChild(emptyTr);
            } else {
                slice.forEach(row => {
                    const tr = el('tr');
                    columns.forEach(col => {
                        const td = el('td', { 'data-key': col.key });
                        if (col.key === 'Nivel de riesgo rural') {
                            td.appendChild(renderRiskMeter(row[col.key]));
                        } else {
                            td.textContent = formatCell(row[col.key], col);
                        }
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
            }

            // header sort arrows
            headerRow.querySelectorAll('th').forEach(th => {
                th.classList.remove('is-sorted-asc', 'is-sorted-desc');
                const arrow = th.querySelector('.datatable__sort-arrow');
                if (!arrow) return;
                if (th.dataset.key === state.sort.key) {
                    th.classList.add(state.sort.dir === 'asc' ? 'is-sorted-asc' : 'is-sorted-desc');
                    arrow.textContent = state.sort.dir === 'asc' ? '↑' : '↓';
                } else {
                    arrow.textContent = '↕';
                }
            });

            // status + pagination
            const total = state.data.length;
            const matched = state.view.length;
            const from = matched === 0 ? 0 : start + 1;
            const to = Math.min(end, matched);
            status.textContent = matched === total
                ? `${total.toLocaleString()} records`
                : `${matched.toLocaleString()} of ${total.toLocaleString()} records match`;
            pageInfo.textContent = `${from.toLocaleString()}–${to.toLocaleString()} · Page ${state.page} of ${totalPages()}`;
            prevBtn.disabled = state.page <= 1;
            nextBtn.disabled = state.page >= totalPages();
        }

        function buildFilterOptions(data) {
            filters.forEach(f => {
                const select = filterSelects[f.key];
                const values = Array.from(new Set(data.map(r => r[f.key]).filter(v => v != null)));
                if (f.order) values.sort((a, b) => f.order.indexOf(a) - f.order.indexOf(b));
                else values.sort((a, b) => compareValues(a, b, f.type));
                values.forEach(v => {
                    select.appendChild(el('option', { value: v }, String(v)));
                });
            });
        }

        // ---- fetch ----
        status.textContent = 'Loading…';
        fetch(opts.dataUrl)
            .then(r => r.json())
            .then(data => {
                state.data = data;
                buildFilterOptions(data);
                applyView();
            })
            .catch(err => {
                status.textContent = 'Could not load dataset.';
                console.error(err);
            });
    }

    global.createDataTable = createDataTable;
})(window);